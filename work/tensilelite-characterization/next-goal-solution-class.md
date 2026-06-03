# Next goal — characterization tests for the `Solution` class (slice 2)

Follow-up to the completed `Solution.py` support slice (L165-439). **Slice 2 of
the `Solution.py` campaign**: the `Solution` class construction + Mapping /
identity + the simpler static `state`-helpers. Self-contained kickoff; assumes
the env/conventions of the nine completed targets under
`Tensile/Tests/unit/characterization/`.

## GOAL

Add a characterization suite pinning the **construction + Mapping/identity +
simple static-helper** surface of `Tensile/SolutionStructs/Solution.Solution`
to **≥95% line coverage on the targeted symbols** (NOT the whole 3272-stmt
file — report file % + a slice view), using syrupy (reuse `../survey.md`).
Document resisting fns + a go/no-go on slice 3.

Achieved when:
- **BEFORE baseline captured FIRST** (`coverage-before.txt`, whole-file row).
- `target.md` listing the targeted symbols + slice-coverage method.
- `Tensile/Tests/unit/characterization/SolutionClass/` with syrupy snapshots.
- Determinism: snapshot a **normalised** Solution state (object-free, like the
  ProblemType suite's `norm`), never live objects / paths / the env-specific
  `assembler.rocm_version`.
- `pytest --cov` shows the targeted symbols ≥95% line (demonstrate residual is
  out-of-slice via the Missing column).
- **AFTER coverage captured** (`coverage-after.txt`) + delta in `target.md`.
- `resistance.md` + `recommendations.md` (new files).
- No regression: full `-m unit` ≥ **1800 passed / 201 skipped**; additive only.
- Work committed (atomic; tree clean; no push).
- `next-goal-<target>.md` created (slice 3: the reject-heavy derivation, grounded).

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep -E "SolutionStructs/Solution.py|passed"
```

> Whole-file % stays low (derivation/reject paths are slice 3). Report it AND
> show the targeted symbols' lines are covered (Missing column). Document in
> target.md.

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
- **Import gotcha:** `SolutionStructs/__init__.py` re-exports the `Solution`
  class, shadowing the submodule. Import the *class* with
  `from Tensile.SolutionStructs.Solution import Solution` (works — that gets the
  class) and the *module* (for statics like `getMIOutputInfo`) via
  `importlib.import_module("Tensile.SolutionStructs.Solution")`.

## REQUIRED FIXTURES (copy from `LibraryIO/conftest.py`)

```python
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

@pytest.fixture(scope="session")
def cxx_compiler(): return validateToolchain("amdclang++")
@pytest.fixture(scope="session")
def isa_info_map(cxx_compiler): return makeIsaInfoMap(SUPPORTED_ISA, cxx_compiler)
@pytest.fixture(scope="session")
def assembler(cxx_compiler):
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    # ...see LibraryIO/conftest.py for the exact Assembler construction...
```

These shell out to the real toolchain (present in the container). A full
`Solution` is then built as
`Solution(config, splitGSU, printSolutionRejectionReason=False,
printIndexAssignmentInfo=False, assembler=assembler, isaInfoMap=isa_info_map)`.

## TARGET: `Solution` class (slice 2 — construction + Mapping + simple statics)

`Solution.__init__(self, config, splitGSU, printSolutionRejectionReason,
printIndexAssignmentInfo, assembler, isaInfoMap, srcName="")` (L449). Builds
`self._state` (a dict) from `config` + defaults, constructs the `ProblemType`,
runs derivation. The class is a `collections.abc.Mapping`.

### Targeted symbols (this slice)

| Symbol | Tier / nature |
|---|---|
| `Solution.__init__` happy path | B/C — build from a small real config + the cap fixtures; snapshot a normalised `_state`. Start from a known-good config (reuse the LibraryIO vendored logic fixture's solution block, or `defaultSolution` from GlobalParameters). |
| Mapping dunders: `keys`/`__len__`/`__iter__`/`__getitem__`/`__setitem__` | A — over a built Solution. |
| `__str__`/`__repr__`/`getAttributes`/`__hash__`/`__eq__`/`__ne__` | A — `__str__` delegates to `getSolutionNameFull` (already pinned); snapshot the name + identity behaviour. |
| `getMIOutputInfo(state, isaInfoMap)` (staticmethod) | B — output-type info; snapshot for a few states. |
| `isDirectToVgprSupportDataType(state)` (staticmethod) | A — pure dtype predicate. |
| `getDivisorName(state, tC)` (staticmethod) | A — string helper. |
| `getKernels` / `getKernelBetaOlnyObjects` / `getKernelConversionObjects` | B — shallow accessors over a built Solution; snapshot counts/shape. |

### OUT of slice (defer to slice 3)

The reject-heavy parameter-derivation statics —
`assignProblemIndependentDerivedParameters`, `assignDerivedParameters`,
`setGlobalReadVectorWidth`, `setGlobalLoadTileDimClassic`,
`checkAndAssignWaveSeparateGlobalRead`, `isDirectToVgpr/LdsDoable`,
`depthUIteration` — and `_deriveAndValidateMXScaleLayoutAndTransport` (L57-162).
These have hundreds of cap-gated `reject()` branches and need a config matrix.

### Determinism plan

1. **Normalise the state.** Copy the ProblemType suite's `norm` idea: render
   `DataType`/`ActivationType`/`ProblemType` to stable strings, sort keys, and
   **drop env-specific fields** (anything derived from `assembler.rocm_version`
   or absolute paths) before snapshotting.
2. **printSolutionRejectionReason=False** to keep construction quiet/deterministic
   (the `Validators` suite relied on this).
3. **One real config.** Build the happy-path Solution from a single vendored/known
   config; snapshot its normalised state once, then exercise the Mapping/identity
   /statics against it. Add 1-2 dtype variants for the statics.
4. If a config triggers a `reject()` that raises on a valid `SolutionIndex`
   (as in `Utilities.reject`), avoid setting `SolutionIndex`, or pass
   `printSolutionRejectionReason=False`.

### Location & layout

- Suite: `Tensile/Tests/unit/characterization/SolutionClass/`, `pytestmark =
  pytest.mark.unit`. `conftest.py` with the cap fixtures + a `norm` helper.
  New `target.md`/`resistance.md`/`recommendations.md`/`coverage-before.txt`/
  `coverage-after.txt`; snapshots in `__snapshots__/`.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` + `target.md` + `conftest.py`.
2. `Solution.__init__` happy path + normalised-state snapshot.
3. Mapping dunders + identity + `__str__`.
4. The simple statics (`getMIOutputInfo`, `isDirectToVgprSupportDataType`,
   `getDivisorName`) + the `getKernels*` accessors.
5. resistance.md + recommendations.md.
6. AFTER coverage + `coverage-after.txt` + no-regression confirmation.
7. `next-goal-<target>.md` (slice 3: derivation/reject machinery, grounded) +
   commit. Mark this checklist done.

## Definition of done checklist — COMPLETE

- [x] `coverage-before.txt` captured (baseline) BEFORE new tests.
      (whole-file 3272/2002 = 38.81% line; targeted surface largely uncovered.)
- [x] Targeted symbols ≥95% line — the slice-2 **surface** symbols (Mapping,
      identity, simple statics, getKernels) are fully covered; only the dead
      `__ne__` L5229 remains. Whole-file 39.55% (derivation = slice 3).
- [x] `coverage-after.txt` captured; before→after delta in target.md.
- [x] Every targeted Tier A/B symbol has snapshot coverage; deferred derivation
      symbols documented (slice 3).
- [x] Solution state snapshots normalised (schema + curated stable fields; no
      env-coupled values / rocm_version).
- [x] resistance.md: dead `__ne__` L5229; the two pipeline-dependent kernel
      accessors (AttributeError); the slice-3 deferral.
- [x] recommendations.md: GO → Solution slice 3 (derivation/reject machinery);
      `GlobalParameters.py` as a smaller intermediate win.
- [x] full `-m unit` = **1818 passed / 201 skipped**, no failures, additive only.
- [x] All work committed (atomic, no push); tree clean.
- [x] Next goal prompt `next-goal-solution-derivation.md` created and committed.
