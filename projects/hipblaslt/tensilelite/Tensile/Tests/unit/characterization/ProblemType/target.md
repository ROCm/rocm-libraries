# Characterization target — `Tensile/SolutionStructs/Problem.py` (ProblemType slice)

Follow-up to the completed `ValidParameters/` suite (#7). Same pattern, but a
**slice** of a large file. Kept as **new files** under the `ProblemType/` test
dir per the add-only rule. See `../survey.md` for the syrupy survey (reused).

## Module under test (the slice)

`Tensile/SolutionStructs/Problem.py` is 1382 LOC / 601 stmts. This target
characterizes the **ProblemType slice** — which is nearly the whole file:

| Symbol | LOC area | Tier |
|---|---|---|
| `getRealDataTypeA` / `getRealDataTypeB` | 377-399 | A — mix-type dtype mappers |
| `_defaultProblemType` / `_expectedProblemTypeParamTypes` | 408-506, 755-757 | A — registries |
| `problemTypeToEnum` | 711-750 | A — in-place DataType→enum converter |
| `validateProblemTypeParameterTypes` | 760-799 | B — type checker into Solution's shared collector |
| `ProblemType` (`Mapping`) | 802-1366 | B/C — construction, GEMM-type check, `initGEMM`, `assignDerivedParameters`, the branchy `__str__`, Mapping dunders, eq/hash |
| `getBiasDataTypeListDefault` | 1372-1382 | A — default bias dtype list |
| `ProblemSizeRange` | 39-171 | B — size-range parsing/enumeration |
| `Problem` / `ExactList` / `ExactDict` | 173-261 | B — exact-problem holders |
| `ProblemSizesMock` / `…Dummy` | 271-277 | A — trivial holders |
| `ProblemSizes` | 279-375 | B — range/exact aggregation + max-size computation |

Imports are **clean** — `rocisa.enum`, `DataType`, `ActivationType`,
`Constants`, `Utilities`; **no** `isaInfoMap`/`asmCaps`/`assembler`/`subprocess`/
toolchain anywhere. So the slice is pure-Python over dicts + the (already
pinned) `DataType` and `ActivationType`.

## Why this module

- **Closes the type/serialize arc.** `DataType` → `ValidParameters` →
  **`ProblemType`** → `LibraryIO` (done). `ProblemType` is the object LibraryIO
  serialises and whose dtype fields `DataType` defines; pinning its construction
  + naming anchors the Solution/Problem core.
- **Pure and high-leverage.** Recommended by the `ValidParameters`
  `recommendations.md`. No toolchain/GPU; the heavy parts are branchy logic, not
  environmental coupling.

## Determinism handling

1. **Object-free state view.** `ProblemType.state` holds live `DataType` /
   `ActivationType` objects; the `conftest.norm` helper renders them to stable
   strings (`"<DataType Float>"`, `"<ActivationType None>"`) and sorts keys.
2. **Minimal (YAML-mirroring) configs.** Configs are minimal dicts (only the
   keys set) so the dtype-derivation `if "X" in config` guards behave as in
   production — starting from the full `_defaultProblemType` would pin
   `MacDataTypeA`/`DataTypeA/B` to 0 and override any `DataType` change.
3. **Isolated shared collector.** `validateProblemTypeParameterTypes` writes a
   module-global collector in `Solution`; tests clear/capture-delta/restore it
   in a `finally`, leaving the session unaffected.
4. **`printExit` is `sys.exit(-1)`** → reject paths pinned via
   `pytest.raises(SystemExit)`.

## Location & coverage command

Suite at `Tensile/Tests/unit/characterization/ProblemType/`, marked `-m unit`.
Pass `--cov` the package dir + grep the `SolutionStructs/Problem.py` row:

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "SolutionStructs/Problem.py"
```

## Result (before → after)

| | Stmts | Miss | Line cov | Blended |
|---|---|---|---|---|
| Before | 601 | 240 | 60.07% | 55.63% |
| After  | 601 | 18 | **97.00%** | 95.61% |

Delta: **+36.93 pts line**, −222 missing statements. The 18 residual line-misses
are all provably unreachable / dead code for any GEMM `ProblemType` built through
the public path (post-first-raise dead branches, the attribute-bug `isGEMM`,
the GEMM-invariant index-validation raises, the dead `UseE` `elif`, the
unreachable `Index01` ordering branch) — full line-by-line accounting in
`resistance.md`.

No regression: full `-m unit` went **1563 → 1672 passed** (+109 new tests),
201 skipped unchanged. Per-row detail in `coverage-after.txt`; next-target
go/no-go in `recommendations.md`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `conftest.py` | `norm` (object-free state view) + `make_pt` (minimal-config builder) |
| `test_helpers_char.py` | `getRealDataType{A,B}`, `problemTypeToEnum`, `validateProblemTypeParameterTypes`, `getBiasDataTypeListDefault`, registries |
| `test_problemtype_char.py` | `ProblemType` across ~42 feature configs + Mapping/eq/hash/`FromDefaultConfig`/`assignDerivedParameters` early-return + raise paths |
| `test_problemsizes_char.py` | `ProblemSizeRange`, `Problem`, `ExactList`/`ExactDict`, `ProblemSizesMock(Dummy)`, `ProblemSizes` |
