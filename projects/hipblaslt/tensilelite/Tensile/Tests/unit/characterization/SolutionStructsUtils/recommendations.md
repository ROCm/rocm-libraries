# Recommendations — after `SolutionStructs/Utilities.py` + `LdsPadding.py`

New file in the `SolutionStructsUtils/` test dir per the add-only rule.

## Result for this target

| Module | Before line | After line |
|---|---|---|
| `Utilities.py` | 53.06% | **100.00%** |
| `LdsPadding.py` | 90.09% | **100.00%** |

`Utilities` reached 100% line and branch; `LdsPadding` reached 100% line (2
residual partial branches — lowest-overhead-tie paths the access patterns never
take). No regression: full `-m unit` **1713 → 1769 passed** (+56), 201 skipped
unchanged. Came in under the ~0.5–1 day estimate.

## What worked (additions to the shared list)

- **Direct private-helper tests for defensive code the public API can't reach.**
  A container probe proved the `LdsPadding` fp16 closed-form never fails for the
  real b128 access pattern, so the search-fallback + bank-reject branches are
  dead via the public selectors. Rather than leave ~17 lines uncovered, the
  private bank-check helpers were tested directly (and the fp16 search forced via
  a monkeypatched `_b128_check`) — characterizing their contract and reaching
  100% line. Documented as such in `resistance.md`.
- **Grid-snapshot numeric solvers.** For pure numeric config functions, a
  parametrised (macro-tile × wave × vector-width) grid with the full result dict
  snapshotted per input pins the tables and walks the tier-search branches with
  little code.

## Go / no-go on the next target

### Verdict: **GO — `SolutionStructs/Solution.py` support-functions slice** (the first slice of the `Solution.py` campaign; defer the cap-coupled `Solution` class + `GlobalParameters.py`)

`Solution.py` is 5230 LOC / 3272 stmts at 32.78% — a multi-slice campaign. But
its **head (L57-439) splits cleanly into a pure slice and a cap-coupled rest**:

| Candidate | Why / why not | Effort to ≥95% line (of the slice) |
|---|---|---|
| **`Solution.py` support-functions slice (L165-439)** ✅ chosen | **Pure** — `_getExpectedTypes`, the type-mismatch collector machinery (`reset`/`get`/`merge`/`validateParameterTypes`/`printTypeMismatchSummary`), the arg dataclasses (`Fbs`, `FactorDimArgs`, `BiasTypeArgs`, `activationSetting`, `ActivationArgs`), and the index helpers (`isPackedIndex`, `isExtractableIndex`). No caps/toolchain; the dataclasses just need a `ProblemType` (now pinned) + a small config. Mirrors the `ProblemType` validate machinery already characterized. Highest coverage-per-effort in `Solution.py`. | ~0.5–1 day |
| `Solution.py` `Solution` class + `_deriveAndValidateMXScaleLayoutAndTransport` | The bulk (~3000 stmts), but heavily cap/toolchain-coupled (104 refs to `asmCaps`/`archCaps`/`assembler`/`isaInfoMap`). Needs the real `isaInfoMap`/`assembler` fixtures (as `LibraryIO`/`Validators` used) and careful slicing (construction → parameter derivation → asm-cap branches). Subsequent slices. | multi-day, multi-slice |
| `Common/GlobalParameters.py` (220 stmts, 84.19%) | Env-coupled (subprocess GPU clocks, `__version__`, mutable globals, `isaInfoMap`); a monkeypatch-heavy top-up. Reasonable, but lower leverage than starting the `Solution.py` campaign. | ~1–1.5 day |

**Why the Solution.py support slice now:** it is the last cheap pure slice and
the natural on-ramp to the `Solution.py` campaign — it shares the type-mismatch
collector with `ProblemType` (already characterized) and the arg dataclasses
build on the pinned `ProblemType`. Bank it, then mount the cap-driven `Solution`
class slices (reusing the established `isaInfoMap`/`assembler` fixtures), and
keep `GlobalParameters` as a later env-coupled top-up.

### Effort estimate

~0.5–1 day. Pure; the collector functions mutate a module global (clear/capture/
restore, as the `ProblemType` validate test did); the arg dataclasses need a
`ProblemType` + small configs and emit `printWarning`/`printExit` on bad input
(pin the `printExit`→`SystemExit` paths). A grounded API inventory + BEFORE
baseline at kickoff is in the companion `next-goal-solution-support.md`.
