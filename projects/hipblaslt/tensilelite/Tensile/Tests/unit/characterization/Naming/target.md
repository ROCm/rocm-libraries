# Characterization target — `Tensile/SolutionStructs/Naming.py`

Follow-up to the completed `ProblemType/` slice (#7). Same pattern, next module.
Kept as **new files** under the `Naming/` test dir per the add-only rule. See
`../survey.md` for the syrupy survey (reused).

## Module under test

`Tensile/SolutionStructs/Naming.py` (239 LOC, 120 stmts) — the kernel/solution
**name builders**. Given a solution `state` dict it produces canonical name
strings used for filenames, dedup keys, and library logic. **Clean imports** —
`functools.lru_cache`, `Common.Constants.MAX_FILENAME_LENGTH`,
`Common.RequiredParameters.getRequiredParametersMin/Full`, `.Problem.ProblemType`;
no toolchain/GPU/subprocess.

| Symbol | LOC | Tier |
|---|---|---|
| `getParameterNameAbbreviation` | 100-102 | A — uppercase-letters abbrev (`lru_cache`) |
| `getPrimitiveParameterValueAbbreviation` | 105-125 | A — per-type abbrev (str/bool/int±/ProblemType/float) |
| `getParameterValueAbbreviation` | 128-141 | A — ISA/scalar/tuple/list/dict (+ dead raise) |
| `_getName` | 144-205 | B — the core builder (option branches) |
| `getKeyNoInternalArgs` | 47-97 | B — dedup key with internal args masked |
| `shortenFileBase` | 208-219 | A/B — base or sha256/base64-shortened |
| `getKernelFileBase` | 222-227 | A — custom vs shortened |
| `getKernelNameMin` / `getSolutionNameMin` / `getSolutionNameFull` | 230-239 | A — `_getName` wrappers |
| **TOTAL** | | **120 stmts** |

## Why this module

- **Cheapest pure module downstream of `ProblemType`** (just pinned). It names
  the very `ProblemType`/solution states the prior target characterized, and has
  **no existing dedicated unit test**. Recommended by the `ProblemType`
  `recommendations.md`.

## Determinism handling

Names are pure functions of the input `state`; snapshot the returned strings.
The solution `state` is built by the `conftest.make_state` factory — a real
`ProblemType` plus the `GlobalSplitU` / internal-args / `SpaceFillingAlgo` /
tile keys that `_getName` and `getKeyNoInternalArgs` read — overridable per
test. `_getName` / `getKeyNoInternalArgs` mutate-then-restore `GlobalSplitU`
and `ProblemType.GroupedGemm`; dedicated tests pin the exact restore.
`shortenFileBase`'s long path is a deterministic sha256+base64 of the name tail.

## Location & coverage command

Suite at `Tensile/Tests/unit/characterization/Naming/`, marked `-m unit`.

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "SolutionStructs/Naming.py"
```

## Result (before → after)

| | Stmts | Miss | Line cov | Blended |
|---|---|---|---|---|
| Before | 120 | 21 | 82.50% | 75.56% |
| After  | 120 | 1 | **99.17%** | 98.89% |

Delta: **+16.67 pts line**, −20 missing statements. The 1 residual line-miss
(L141) is a provably-unreachable defensive `raise` (a composite value always
matches `tuple`/`list`/`dict` first). The suite also **characterized a latent
bug**: `getKernelNameMin(splitGSU=True)` with `GlobalSplitU > 1` or `== -1`
raises `TypeError` (`"M" > 0` at L160) — pinned via `pytest.raises`; details in
`resistance.md`.

No regression: full `-m unit` went **1672 → 1713 passed** (+41 new tests),
201 skipped unchanged. Per-row detail in `coverage-after.txt`; next-target
go/no-go in `recommendations.md`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `conftest.py` | `make_state` solution-state factory |
| `test_naming_char.py` | every abbrev type branch, `_getName` option branches via the name wrappers, `getKeyNoInternalArgs`, `shortenFileBase`/`getKernelFileBase` |
