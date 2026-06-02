# Characterization target — `Tensile/TensileLogic/`

Follow-up to the completed `SolutionStructs/Validators/` suite (#7). Same
pattern, next module. Kept as a **new file** under the `TensileLogic/` test
dir rather than editing the shared `../target.md`, per the add-only rule.
See `../survey.md` for the syrupy survey (reused, no new survey needed).

## Module under test

Eight source files — the `TensileLogic` CLI driver, which validates
serialized library-logic YAML before it ships. `_runChecks` walks a logic
tree and runs the per-solution validators (`_validateMatrixInstruction`,
`_validateWorkGroup`, `_validateWorkGroupMappingXCC`) plus file-level
`_validateChipId`; `main`/`_setup` wire up argparse + toolchain + the
`ParallelMap2` fan-out.

| File | Stmts | Baseline cov | Public API (tier) |
|---|---|---|---|
| `ValidWorkGroup.py` | 11 | 0% | `_validateWorkGroup` (A) |
| `ValidMatrixInstruction.py` | 12 | 0% | `_validateMatrixInstruction` (A) |
| `ValidWorkGroupMappingXCC.py` | 39 | 0% | `_validateWorkGroupMappingXCC`, `reset_reported_failures`, `_report_xcc_failure`, `_cu_count_from_path` (A/B) |
| `KnownBugs.py` | 43 | 76% | `normalize_logic_relative_path`, `load_known_bugs`, `is_known_bug` (B) |
| `ValidChipId.py` | 92 | 95% | `_validateChipId` + helpers (B) |
| `HandleCustomKernel.py` | 47 | 0% | `handleCustomKernel`, `hasCustomKernel`, `prepareCustomKernelConfig` (B) |
| `ParseArguments.py` | 15 | 0% | `parseArguments` (B) |
| `Run.py` | 133 | 0% | `_runChecks`, `_setup`, `_progress_loop`, `main`, `Check` (C) |
| `__init__.py` | 1 | 0% | (re-exports `main`) |
| **TOTAL** | **393** | **31.55% line** | |

Baseline rows captured in `coverage-before.txt` (headline: 393 stmts,
269 missing → **31.55% line / 32.89% blended w/ branch**).

## Why this module

- **Natural next step from `Validators/`.** `TensileLogic` is the *caller*
  of the validators just characterized — `_runChecks` drives
  `_validateMatrixInstruction` / `_validateWorkGroup` over real logic files.
  Pinning it closes the loop from "validator behaviour" to "what the CLI
  actually does with a logic tree".
- **Cheap tiers dominate.** Five of eight files are pure or near-pure
  (string/set logic, argparse, table lookups). Only `Run.py` has real
  statefulness (threading, subprocess, multiprocessing) — and most of *that*
  is tractable by injecting the validators / `isaInfoMap` it already imports.
- **Partial tests prove the shape works.** `ValidChipId` (95%) and
  `KnownBugs` (76%) are already exercised via importlib in the existing unit
  tests; the characterization suite fills their remaining error branches and
  adds the seven untouched files.

## Determinism handling (more surface than `Validators/`)

1. **Filesystem paths.** Validators derive behaviour/messages from a
   `filepath`. Inputs use stable synthetic paths
   (`Path("logic/asm/aquavanjaram/X.yaml")`) or `tmp_path`, never real
   absolute paths in snapshots.
2. **Module-global state.** `ValidWorkGroupMappingXCC._xcc_failures_by_file`
   accumulates across calls. Every XCC test calls `reset_reported_failures()`
   first and snapshots `{returned, reported_failures}` (state + return), so
   results never depend on test order.
3. **YAML config.** `KnownBugs.load_known_bugs` reads a file → `tmp_path`
   fixtures; the resulting frozenset is **sorted** before snapshotting.
4. **threading / time / subprocess (`Run.py`).** `_progress_loop` (threads +
   `time.time`) and `_setup`/`main` (subprocess + `ParallelMap2`) are
   exercised for line coverage only, never snapshotted. `_runChecks`/`main`
   are characterized with the imported validators / `_setup` / `ParallelMap2`
   monkeypatched in the `Run` namespace, so the orchestration logic (keep /
   total / known-bug-skip / chip-id-failure counts, batching, exit codes) is
   the thing pinned — deterministically, with no live toolchain or fan-out.

## Location & coverage command (same rules as `Validators/`)

Suite at `Tensile/Tests/unit/characterization/TensileLogic/`, marked
`-m unit` (collected by the existing `testpaths=Tensile/Tests`, no config
edit). Pass `--cov` a **directory path**, never a dotted module (a dotted
name re-imports rocisa → SIGABRT):

```
pytest -m unit \
  --cov=Tensile/TensileLogic \
  --cov-config=pyproject.toml --cov-report=term-missing \
  Tensile/Tests/unit
```

Line coverage = `(Stmts - Miss) / Stmts` (the goal's bar is line; both line
and blended are reported).

## Result (before → after)

| | Stmts | Miss | Line cov |
|---|---|---|---|
| Before | 393 | 269 | 31.55% |
| After  | _see `coverage-after.txt`_ | | |

Delta and per-file detail recorded in `coverage-after.txt` and the
`TensileLogic` sections of `../resistance.md`-style `resistance.md` /
`recommendations.md` (kept as new files in this dir).
