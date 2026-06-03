# Characterization target — `Solution` class (slice 2 of the Solution.py campaign)

Follow-up to the completed `SolutionSupport/` slice (#7). **Slice 2**: the
`Solution` class construction + Mapping/identity + simple static helpers. New
files under `SolutionClass/` per the add-only rule. See `../survey.md` (reused).

## Slice under test

`Tensile/SolutionStructs/Solution.Solution` — a `collections.abc.Mapping`
(L444-5230). This slice targets the **class surface**: construction (a real
`Solution` parsed from the vendored logic fixture), the Mapping interface,
identity/hash/equality, and the simpler static `state`-helpers. The
reject-heavy parameter-derivation statics are **slice 3**.

| Targeted symbol | LOC | Tier |
|---|---|---|
| `Solution.__init__` (happy-path construction) | 449-1090 | C — built from a real config via the cap fixtures |
| `getKernels` | 534-543 | A |
| `getKernelBetaOlnyObjects` / `getKernelConversionObjects` | 544-553 | A — pinned as AttributeError (pipeline state) |
| `getMIOutputInfo` (staticmethod) | 554-560 | B |
| `isDirectToVgprSupportDataType` (staticmethod) | 1095-1103 | A |
| `getDivisorName` (staticmethod) | 1402-1414 | A |
| Mapping dunders `keys`/`__len__`/`__iter__`/`__getitem__`/`__setitem__` | 5185-5199 | A |
| `__str__`/`__repr__`/`getAttributes`/`__hash__`/`__eq__`/`__ne__` | 5201-5230 | A |

**Out of slice (slice 3):** `assignProblemIndependentDerivedParameters`,
`assignDerivedParameters`, `setGlobalReadVectorWidth`,
`setGlobalLoadTileDimClassic`, `checkAndAssignWaveSeparateGlobalRead`,
`isDirectToVgpr/LdsDoable`, `depthUIteration`, and
`_deriveAndValidateMXScaleLayoutAndTransport` — the reject/cap-heavy derivation.

## Determinism handling

- A real `Solution` is built by parsing the LibraryIO vendored logic fixture
  (`../LibraryIO/data/logic_gfx942_HSS_BH.yaml`, read-only) via the reused cap
  fixtures (`cxx_compiler`/`isa_info_map`/`assembler`) — avoiding hand-authoring
  a self-consistent kernel.
- The 306-key state carries toolchain-derived values, so the construction
  snapshot pins the **schema** (sorted key set) + a curated set of **stable**
  fields (`KernelLanguage`, `MacroTile*`, `MatrixInst*`, ...) rendered
  object-free, not every value.
- **Import gotcha:** `SolutionStructs/__init__.py` re-exports the `Solution`
  *class*; `from ...Solution import Solution` correctly yields the class.
- `__setitem__` mutates the (session-shared) Solution; the round-trip test
  saves/restores the key.

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "SolutionStructs/Solution.py"
```

## Result (before → after)

| | Whole-file line | Targeted slice-2 surface |
|---|---|---|
| Before | 38.81% | largely uncovered |
| After  | 39.55% | **fully covered** (only L5229 — a dead `__ne__` branch — remains) |

The whole-file rises only modestly because the existing suite already
constructs solutions and the reject-heavy derivation (slice 3) dominates the
remaining misses. Every targeted **surface** symbol is covered except the
provably-dead `__ne__` NotImplemented arm (L5229) — see `resistance.md`.

No regression: full `-m unit` went **1800 → 1818 passed** (+18 new tests),
201 skipped unchanged. Per-row detail in `coverage-after.txt`; next-slice
go/no-go in `recommendations.md`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `conftest.py` | cap fixtures + real `solutions`/`solution` (from the vendored fixture) + `solution_summary` |
| `test_solution_class_char.py` | construction, Mapping, identity, the simple statics, `getKernels` |
