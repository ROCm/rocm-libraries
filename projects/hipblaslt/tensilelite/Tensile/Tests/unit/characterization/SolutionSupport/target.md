# Characterization target — `Solution.py` support-functions slice (L165-439)

Follow-up to the completed `SolutionStructsUtils/` suite (#7). The **first slice
of the `Solution.py` campaign** — its pure module-level head. New files under
`SolutionSupport/` per the add-only rule. See `../survey.md` (reused).

## Slice under test

`Tensile/SolutionStructs/Solution.py` is 5230 LOC / 3272 stmts — a multi-slice
target. This slice is the **pure support head (L165-439)**, before the
cap-coupled `Solution` class (L444+):

| Symbol | LOC | Tier |
|---|---|---|
| `_getExpectedTypes` + `_expectedParamTypes` + `_skipTypeCheck` | 165-201 | A — type registry from `validParameters` |
| `resetTypeMismatchCollector` / `getTypeMismatchCollector` / `mergeTypeMismatchCollector` | 211-245 | A/B — the shared mismatch collector |
| `validateParameterTypes` | 248-285 | B — type checker into the collector |
| `printTypeMismatchSummary` | 288-339 | B — stdout summary |
| `Fbs` (Enum) | 342-345 | A |
| `FactorDimArgs` / `BiasTypeArgs` / `activationSetting` / `ActivationArgs` | 350-423 | B — benchmark-arg dataclasses |
| `isPackedIndex` / `isExtractableIndex` | 427-439 | A — index helpers |

`_deriveAndValidateMXScaleLayoutAndTransport` (L57-162) and the `Solution` class
(L444+) are **out of slice** (cap/toolchain-coupled — subsequent slices).

## Why this slice

Recommended by the `SolutionStructsUtils` `recommendations.md`: the last cheap
**pure** slice and the on-ramp to the `Solution.py` campaign. It shares the
type-mismatch collector with `ProblemType` (already characterized) and the arg
dataclasses build on the now-pinned `ProblemType`.

## Determinism handling

- The collector is a module global shared with `ProblemType` validation; an
  `isolated_collector` fixture clears it, yields, and restores the prior
  contents in a `finally` (snapshots see only the per-test delta; session safe).
- The arg dataclasses take a `ProblemType` (built minimal via the `make_pt`
  fixture) + a small config; `printExit`→`sys.exit(-1)` paths pinned via
  `pytest.raises(SystemExit)`; `printWarning` lines tolerated; `capsys` captures
  `printTypeMismatchSummary` stdout.
- **Import note:** `SolutionStructs/__init__.py` re-exports the `Solution`
  *class*, shadowing the submodule attribute, so the module is loaded via
  `importlib.import_module("Tensile.SolutionStructs.Solution")`.

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "SolutionStructs/Solution.py"
```

Slice target: the whole-file % stays low (the `Solution` class is out of slice);
slice coverage is demonstrated via the Missing column (no missed lines in
165-439 — the first miss is L473, inside the out-of-slice class).

## Result (before → after)

| | Whole-file line | Slice (L165-439) |
|---|---|---|
| Before | 37.50% | targeted symbols largely uncovered |
| After  | 38.81% | **100% line** (0 missing lines / 0 partial branches in-slice) |

The whole-file delta is small by design — only the L165-439 slice was targeted;
the cap-coupled `Solution` class (~3000 stmts) is the next campaign slice. The
AFTER Missing column's first entry is L473, confirming the slice is fully
covered.

No regression: full `-m unit` went **1769 → 1800 passed** (+31 new tests),
201 skipped unchanged. Per-row detail in `coverage-after.txt`; next-target
go/no-go in `recommendations.md`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `conftest.py` | `make_pt` (minimal ProblemType) + `isolated_collector` |
| `test_solution_support_char.py` | the full L165-439 slice |
