# Characterization target — `SolutionStructs/Utilities.py` + `LdsPadding.py`

Follow-up to the completed `Naming/` suite (#7). **One grouped target** covering
the two remaining pure `SolutionStructs` modules. New files under
`SolutionStructsUtils/` per the add-only rule. See `../survey.md` (reused).

## Modules under test

| Module | LOC | Stmts | Role |
|---|---|---|---|
| `SolutionStructs/Utilities.py` | 113 | 49 | small pure helpers: `getMiInputType`, `reject`, `pvar`, `roundupRatio`, `getRealDataType{A,B}` |
| `SolutionStructs/LdsPadding.py` | 412 | 212 | pure numeric LDS-padding solvers: `get_fp4/fp8/fp16/fp32/mxs_mt_config` + `_compute_*` / bank-conflict-check helpers |

Both **pure** — `Utilities` imports `sys`/`math`/`DataType`/`rocisa.enum`;
`LdsPadding` imports `functools`/`typing`. No toolchain/GPU/subprocess.

## Why this module

Recommended by the `Naming` `recommendations.md`: the two cheapest remaining
pure `SolutionStructs` modules. `Utilities` was barely tested (53% line);
`LdsPadding` had a partial `test_LdsPadding.py` (90% line). Banking both
finishes the pure `SolutionStructs` surface before the big sliced `Solution.py`
campaign.

## Determinism handling

All outputs are pure ints / strings / `DataType`s — snapshot directly.
- `Utilities.reject` mutates `state` and prints; driven with
  `printSolutionRejectionReason=False` for the quiet paths and snapshotting the
  return + `state["Valid"]`; the valid-`SolutionIndex` raise is pinned via
  `pytest.raises`.
- `LdsPadding` selectors are exercised over a (macro-tile × wave × vector-width)
  grid; the full result config (`perBlock`/`pad`/`shift`) is snapshotted per
  input. The defensive bank-conflict reject branches and the fp16 search
  fallback are unreachable for the real access patterns (the closed-form / tiered
  search always picks a passing config), so they are characterized by **direct
  tests of the private helpers** (and the fp16 search forced via a monkeypatched
  bank check) — see `resistance.md`.

## Coverage command

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit \
  | grep -E "SolutionStructs/(Utilities|LdsPadding).py"
```

## Result (before → after)

| Module | Stmts | Before line | After line | Δ |
|---|---|---|---|---|
| `Utilities.py` | 49 | 53.06% | **100.00%** | +46.94 |
| `LdsPadding.py` | 212 | 90.09% | **100.00%** | +9.91 |

Both reach 100% line. `LdsPadding` has 2 residual partial branches
(`122->exit`, `309->311`) — lowest-overhead-tie paths the access patterns never
take; line coverage is 100%.

No regression: full `-m unit` went **1713 → 1769 passed** (+56 new tests),
201 skipped unchanged. Per-row detail in `coverage-after.txt`; next-target
go/no-go in `recommendations.md`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `test_utilities_char.py` | `getMiInputType`, `reject`, `pvar`, `roundupRatio`, `getRealDataType{A,B}` |
| `test_ldspadding_char.py` | `get_fp4/fp8/fp16/fp32/mxs_mt_config` over an input grid + direct private bank-check / search-fallback tests |
