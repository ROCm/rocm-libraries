# Next goal — characterization tests for `Tensile/TensileLogic/`

Follow-up to the completed `SolutionStructs/Validators/` suite (#7). Same
pattern, next module. This file is the self-contained kickoff; it assumes the
env and conventions already established (see `prompt.md`, `env/README.md`,
`HANDOFF.md`).

## GOAL

Add a characterization-test suite that pins current behaviour of
`Tensile/TensileLogic/` with **≥95% line coverage on the module**, using
syrupy (already chosen — reuse `Tensile/Tests/unit/characterization/survey.md`,
no new survey needed), integrated with the existing pytest/tox setup.
Document resisting functions + a go/no-go on the *next* target after this one.

Achieved when:
- `target.md` for TensileLogic (rationale + the location/scope notes below).
- `Tensile/Tests/unit/characterization/TensileLogic/` exists with syrupy
  snapshots driving every reasonably-testable public function (see API
  inventory) with representative inputs.
- Determinism handled: filesystem paths normalised in snapshots; the
  module-global failure-report state reset between cases; no RNG/time/thread
  nondeterminism leaks into snapshots.
- `pytest --cov` (path-mode) reports **≥95% line** on the module.
- `resistance.md` (append a TensileLogic section, or a new file in the
  TensileLogic dir) lists every function that resisted + reason + workaround.
- `recommendations.md` updated: go/no-go on the *following* target (LibraryIO
  vs Common) + effort estimate.
- No regression: full `-m unit` suite still passes (current baseline incl.
  Validators suite = **1249 passed / 201 skipped**; this work only adds).

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**: only add new files. Never modify/delete any existing file
  (source, tests, config, docs). Anything needing an edit → new file or
  document as a limitation.
- **Commit frequently**, atomic commits. **Never push / no PRs.** Local only.
- rocm-libraries boundary: stay within `projects/hipblaslt/tensilelite`.
- Snapshot structured/normalised forms, never raw nondeterministic blobs
  (paths/timestamps/IDs normalised).
- BOUND: 180 turns / 180 min. AUTONOMY: don't ask for confirmation; document
  blockers and continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree mounted at
  /work). rocisa built (needs `LD_LIBRARY_PATH=/opt/rocm/lib`, baked).
- Resume: see `HANDOFF.md` "Resume steps".
- **Coverage gotcha (critical, same as before): pass `--cov` a directory
  PATH, never a dotted module** — a dotted name re-imports rocisa (nanobind)
  → SIGABRT. Canonical run:
  ```
  pytest -m unit \
    --cov=Tensile/TensileLogic \
    --cov-config=pyproject.toml --cov-report=term-missing \
    Tensile/Tests/unit
  ```
- Line coverage = `(Stmts - Miss)/Stmts` from the report (pyproject sets
  `branch=True`; the goal's bar is **line** coverage — report both).

## TARGET: `Tensile/TensileLogic/` (8 files, ~989 LOC)

Existing unit tests already touch: `test_KnownBugs.py`, `test_ValidChipId.py`
(and `test_MatrixInstructionConversion.py` indirectly). rocisa is pulled in
transitively (via the `SolutionStructs/Validators` imports), so it must be
built — it is.

### Public API inventory (drive each; note the tier)

| File | LOC | Public fns | Tier / nature |
|---|---|---|---|
| `ValidWorkGroup.py` | 47 | `_validateWorkGroup(solution, filepath)` | A — thin wrapper over `Validators.validateWorkGroup`; pure + a filepath for messages |
| `ValidMatrixInstruction.py` | 72 | `_validateMatrixInstruction(...)` | A — wraps `Validators.validateMIParameters`; reuse the consistent-solution trick from the Validators suite |
| `ValidWorkGroupMappingXCC.py` | 90 | `_validateWorkGroupMappingXCC(solution, filepath)`, `reset_reported_failures()`, `_report_xcc_failure(...)`, `_cu_count_from_path(filepath)` | A/B — **module-global reported-failures set**; call `reset_reported_failures()` per test; `_cu_count_from_path` parses the path string |
| `ValidChipId.py` | 207 | `_validateChipId(...)` + helpers (`_chipIdKey`, `_archChipIds`, `_sourceChipIds`, `_defaultChipIds`, `_fallbackFamily`, `_chipIdDirFromPath`, `_validateChipIdPlacement`, `_reportChipIdFailure`) | B — pure string/set logic over chip ids, but **derives from filepath**; normalise paths in snapshots; partial tests exist |
| `KnownBugs.py` | 112 | `normalize_logic_relative_path(path)`, `load_known_bugs(config_path)`, `is_known_bug(...)` | B — `load_known_bugs` reads a YAML file (use a tmp file / the repo's `known_bugs.yaml`); `normalize_logic_relative_path` is the path-normaliser; partial tests exist |
| `HandleCustomKernel.py` | 123 | `handleCustomKernel(sol, isaInfoMap)`, `hasCustomKernel(file)`, `prepareCustomKernelConfig(miParams, prevMi)` | B — `prepareCustomKernelConfig` pure (string build); `hasCustomKernel`/`handleCustomKernel` read custom-kernel files (fixtures or tmp) |
| `ParseArguments.py` | 89 | `parseArguments()` | B — argparse; drive with explicit argv lists, snapshot the parsed Namespace (normalise any abs paths) |
| `Run.py` | 248 | `main()`, `_runChecks(...)`, `_setup()`, `_progress_loop(...)`, `Check` | C — orchestration: `validateToolchain`+`makeIsaInfoMap` (subprocess), `threading`, `time`, fs. Characterize `_runChecks` with injected checks if feasible; `main`/`_setup`/`_progress_loop` likely RESIST → resistance.md |

### Determinism plan (more surface than Validators — read these first)

1. **Filesystem paths.** Several validators take a `filepath: Path` and derive
   behaviour or messages from it. Snapshots must **normalise** absolute paths
   to repo-relative (there is already `normalize_logic_relative_path` /
   `_chipIdDirFromPath` — use/observe them). Build inputs with stable
   synthetic paths (e.g. `Path("logic/asm/aquavanjaram/X.yaml")`), not real
   absolute paths.
2. **Module-global state.** `ValidWorkGroupMappingXCC` accumulates reported
   failures in a module global; `_reportChipIdFailure` may do similar. Call
   `reset_reported_failures()` (and any ChipId reset) at the start of each
   test so snapshots don't depend on test order. Per the goal, **snapshot the
   state**, not just the return — capture `{returned, reported_failures}`.
3. **YAML config.** `KnownBugs.load_known_bugs(config_path)` reads a file —
   point it at a tmp YAML fixture (new file) or the repo's `known_bugs.yaml`;
   snapshot the resulting frozenset (sorted for stability).
4. **threading/time/subprocess (Run.py).** `_progress_loop` uses threads +
   time; `_setup` shells out (toolchain). Do **not** snapshot these; either
   test `_runChecks` with injected `Check`s and a prebuilt `isaInfoMap`
   (reuse the session-level `makeIsaInfoMap` like the Validators suite), or
   document `main`/`_setup`/`_progress_loop` as resistance.

### Location & layout (decided, consistent with Validators)

- Suite: `Tensile/Tests/unit/characterization/TensileLogic/`, files marked
  `pytestmark = pytest.mark.unit`. Collected by the existing
  `testpaths=Tensile/Tests` — no config edit (add-only safe).
- `target.md` in `Tensile/Tests/unit/characterization/` (alongside the shared
  `survey.md`). Snapshots in `TensileLogic/__snapshots__/` via
  `--snapshot-update`.
- Reuse the Validators-suite techniques: syrupy `snapshot` fixture;
  `copy.deepcopy(defaultSolution)`; consistent solutions via
  `matrixInstructionToMIParameters`; `printRejectionReason=False`; minimal
  dicts for early-exit/reject branches.

## Suggested commit sequence (atomic)

1. `target.md` for TensileLogic.
2. `ValidWorkGroup` + `ValidMatrixInstruction` suites (cheapest; reuse
   Validators fixtures).
3. `KnownBugs` suite (YAML load + normalise + is_known_bug).
4. `ValidChipId` suite (path-derived placement logic).
5. `ValidWorkGroupMappingXCC` suite (reset global per test; snapshot state).
6. `HandleCustomKernel` + `ParseArguments` suites.
7. `_runChecks` (Run.py) if tractable.
8. resistance.md + recommendations.md updates.
9. Final coverage run (path-mode) + no-regression confirmation.

## Definition of done checklist

- [ ] `--cov=Tensile/TensileLogic` (path-mode) ≥95% line.
- [ ] Every Tier A/B public fn has snapshot coverage; Tier C documented.
- [ ] Module-global state reset per test; paths normalised in snapshots.
- [ ] resistance.md lists each resisting fn (expect: `main`, `_setup`,
      `_progress_loop`, possibly path-coupled message lines).
- [ ] recommendations.md: go/no-go on the target after this (LibraryIO/Common).
- [ ] full `-m unit` ≥ 1249 passed / 201 skipped, no failures, additive only.
