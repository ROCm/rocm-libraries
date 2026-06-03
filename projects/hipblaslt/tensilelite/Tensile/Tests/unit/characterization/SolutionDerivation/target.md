# Characterization target — `Solution` derivation statics (slice 3 of the Solution.py campaign)

Follow-up to the completed `SolutionClass/` slice (#7). **Slice 3a**: the
parameter-derivation statics. New files under `SolutionDerivation/` per the
add-only rule. See `../survey.md` (reused).

## Slice under test

The derivation `@staticmethod`s of `Tensile.SolutionStructs.Solution.Solution`
(plus the two `assign*` entry points). All take a mutable `state` dict (+
`isaInfoMap` and/or `printRejectionReason`) and call `reject()` on invalid
configs.

| Static | LOC | This slice covers |
|---|---|---|
| `setGlobalReadVectorWidth` | 922-940 | **all branches** (crafted states) |
| `checkAndAssignWaveSeparateGlobalRead` | 1062-1069 | **all branches** (crafted states) |
| `getMIOutputInfo` | 554-571 | MFMA path (f32/f64); WMMA arms cap-gated → resist |
| `isVgprForLocalReadPackingDoable` | 1075-1090 | all except the `HasEccHalf=False` arm (gfx942 has it) |
| `assignProblemIndependentDerivedParameters` | 576-919 | early-return guard + full happy re-run + SIA=4 reject |
| `assignDerivedParameters` | 1419-2487 | preamble + early guard + full happy re-run |
| `isDirectToVgprDoable` | 1103-1270 | real-state outcome (A/B) + early rejects |
| `isDirectToLdsDoable` | 1275-1399 | real-state outcome (A/B) + subtile/early rejects |

## Determinism handling

- The small statics are driven with **crafted minimal states** (they read only a
  handful of keys), one mutation per reject branch — deterministic, no fixtures.
- The predicates and the `assign*` methods are seeded from a **real
  fully-derived solution state** (`real_state`, a deep copy of the vendored
  gfx942 HSS_BH solution's `_state`); the `assign*` re-runs reset the
  `Assigned*` flags. Snapshots pin selected derived scalars + reject outcomes
  (object-free), never env-coupled values.
- **Import gotcha:** the module is loaded via `importlib.import_module(...)`
  (the package re-exports the `Solution` class, shadowing the submodule).

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "SolutionStructs/Solution.py"
```

## Result (before → after)

| | Whole-file line |
|---|---|
| Before | 38.81% |
| After  | **41.14%** (+2.33 pts, +52 lines) |

The small statics are fully covered; `getMIOutputInfo`/`isVgprForLocalReadPacking
Doable` reach their gfx942-reachable branches (WMMA/EccHalf arms need other
ISAs); the two giant `assign*` methods and the deep DTV/DTL reject matrices are
**partially** covered (happy path + early/reachable rejects) — the exhaustive
dtype/ISA/MI config sweep is **slice 3b** (see `resistance.md` /
`recommendations.md`).

No regression: full `-m unit` went **1818 → 1844 passed** (+26 new tests),
201 skipped unchanged. Per-static detail in `coverage-after.txt`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `conftest.py` | cap fixtures + `real_state` (deep copy of a real solution state) |
| `test_derivation_char.py` | the derivation statics (small ones fully; predicates + `assign*` partially) |
