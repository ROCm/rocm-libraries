# Next goal — characterization tests for `Tensile/Common/GlobalParameters.py`

Follow-up to the completed `Solution` derivation slice 3a. A **self-contained
top-up** (90% → ≥95% line) chosen over the multi-day Solution slice-3b config
sweep. Self-contained kickoff; assumes the env/conventions of the eleven
completed targets under `Tensile/Tests/unit/characterization/`.

## GOAL

Add a characterization suite raising `Tensile/Common/GlobalParameters.py` to
**≥95% line coverage on the module**, using syrupy (reuse `../survey.md`).
Document resisting fns + a go/no-go on the next target.

Achieved when:
- **BEFORE baseline captured FIRST** (`coverage-before.txt`).
- `target.md` (new file in the test dir).
- `Tensile/Tests/unit/characterization/GlobalParameters/` with syrupy snapshots.
- Determinism: the module mutates a process-global `globalParameters` dict and
  shells out in `setupRestoreClocks`; **save/restore the global** around each
  test and **monkeypatch `subprocess`** for the clock path.
- `pytest --cov` (path-mode + grep) reports **≥95% line** on the module.
- **AFTER coverage** (`coverage-after.txt`) + delta in `target.md`.
- `resistance.md` + `recommendations.md` (new files).
- No regression: full `-m unit` ≥ **1844 passed / 201 skipped**; additive only.
- Work committed (atomic; tree clean; no push).
- `next-goal-<target>.md` created (e.g. Solution slice 3b, or a remaining pure
  module — grounded).

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "Common/GlobalParameters.py|passed"
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
- For `printCapabilitiesTable`/`assignGlobalParameters` you need the real
  `isaInfoMap` — copy the `cxx_compiler`/`isa_info_map` fixtures from
  `LibraryIO/conftest.py`.

## TARGET: `Tensile/Common/GlobalParameters.py` (220 stmts, baseline ~90% line)

Imports `subprocess`/`time`/`os.path`/`__version__`; mutates a process-global
`globalParameters` dict (also `defaultSolution`/`defaultProblemType` built at
import). Current missing lines: **266, 643-695 (assignGlobalParameters
branches), 720-723, 759-763 (setupRestoreClocks)**.

| Symbol | LOC | Tier / nature |
|---|---|---|
| `restoreDefaultGlobalParameters()` | 582-595 | A — resets the global dict; save/restore around the test, snapshot a few keys. |
| `printCapabilitiesTable(isaInfoMap)` | 596-632 | B — prints a caps table; pass the real `isaInfoMap`, capture stdout via `capsys`, snapshot a normalised form (it embeds caps — snapshot structure/headers, not volatile values). |
| `assignGlobalParameters(config, isaInfoMap)` | 633-752 | B — merges a `config` dict into `globalParameters` with per-key branches (L643-695 are the uncovered dispatch arms). Drive with a few config dicts toggling the relevant keys; save/restore the global. |
| `setupRestoreClocks()` | 753-766 | B — shells out (`subprocess`) to read/set GPU clocks (L720-723, 759-763). **Monkeypatch `subprocess.run`/`Popen`** to pin both success and failure paths without a GPU. |
| line 266 | 266 | A — an isolated branch (inspect: likely a one-off conditional). |

### Determinism plan

1. **Process-global isolation.** `globalParameters` is module-global; in each
   test save a deep copy, run, snapshot the delta / selected keys, restore in a
   `finally` (the LibraryIO/ProblemType collector pattern).
2. **subprocess.** Monkeypatch `GlobalParameters.subprocess.run` (and/or
   `Popen`) to return canned output for `setupRestoreClocks`; pin both the
   success parse and the failure/exception arm.
3. **__version__ / time.** If any snapshot would embed `__version__` or a
   timestamp, normalise it (`<VERSION>` / drop), as the LibraryIO suite did.
4. **isaInfoMap** is stable for a given ISA+toolchain in the container.

### Location & layout

- Suite: `Tensile/Tests/unit/characterization/GlobalParameters/`, `pytestmark =
  pytest.mark.unit`. `conftest.py` with the cap fixtures + a global-isolation
  helper. New `target.md`/`resistance.md`/`recommendations.md`/
  `coverage-before.txt`/`coverage-after.txt`; snapshots in `__snapshots__/`.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` + `target.md` + `conftest.py`.
2. `restoreDefaultGlobalParameters` + `printCapabilitiesTable`.
3. `assignGlobalParameters` (config matrix) + the L266 branch.
4. `setupRestoreClocks` (monkeypatched subprocess, success + failure).
5. resistance.md + recommendations.md.
6. AFTER coverage + `coverage-after.txt` + no-regression confirmation.
7. `next-goal-<target>.md` (grounded) + commit. Mark this checklist done.

## Definition of done checklist

- [ ] `coverage-before.txt` captured (baseline) BEFORE new tests.
- [ ] `--cov=Tensile` + grep `GlobalParameters.py` row ≥95% line.
- [ ] `coverage-after.txt` captured; before→after delta in target.md.
- [ ] Every Tier A/B symbol has snapshot coverage; any residue documented.
- [ ] Global dict isolated (save/restore); `subprocess` monkeypatched.
- [ ] resistance.md lists each resisting fn.
- [ ] recommendations.md: go/no-go on the next target (Solution slice 3b vs a
      remaining pure module).
- [ ] full `-m unit` ≥ 1844 passed / 201 skipped, no failures, additive only.
- [ ] All work committed (atomic, no push); tree clean.
- [ ] Next goal prompt `next-goal-<target>.md` created and committed.
