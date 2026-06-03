# Recommendations — after `Tensile/Common/ValidParameters.py`

New file in the `ValidParameters/` test dir per the add-only rule. Builds on the
`DataType/` "GO → ValidParameters next" verdict and the running cost model.

## Result for this target

`Tensile/Common/ValidParameters.py` went from **80.47% line (169 stmts, 33
missing)** to **100.00% line and branch (169 stmts, 0 missing, 0 partial)** —
see `coverage-after.txt`. The existing unit suite already drove the `makeValid*`
builders and the `validParameters` dict (they run during Solution/parameter
validation), so the whole baseline gap was the three tail validators
(L1078–1136); the new validator suite closes it. **Nothing resisted** — see
`resistance.md`.

This came in under the ~1–1.5 day estimate (a few hours): the module is pure
(imports only `math`/`functools`/`SUPPORTED_ISA`/`IsaVersion`), and the only
real care item was summarising the huge builder outputs (the master MI list is
~746k entries) so the snapshots stay reviewable. Suite is 2 files / 37 tests /
26 snapshots, <1s standalone.

No regression: full `-m unit` went **1526 → 1563 passed** (+37), 201 skipped
unchanged — purely additive.

## What worked (additions to the shared list)

- **Summarise huge pure outputs; the loop still runs.** A builder's
  construction loop executes fully when called, so snapshotting a normalised
  `{len, head, tail}` (or per-key `{len, head}`) summary instead of the full
  multi-hundred-thousand-entry list costs zero coverage and keeps the `.ambr`
  reviewable. Document the summary shape in `target.md`.
- **Synthetic `validParams` to isolate validator branches.** Driving
  `checkParametersAreValid` with tiny purpose-built dicts (`{"P": -1}` for the
  any-value accept, `{"P": list(range(40))}` for the `>32`-combos message
  variant, `{"SpaceFillingAlgo": -1}` to reach the sub-validator dispatch) hits
  every branch deterministically without depending on the 138-entry real table.
- **Pin behaviour, not giant interpolated strings.** The unknown-name raise
  interpolates all 138 parameter names; snapshotting only the first message
  line pins the behaviour without coupling the snapshot to the full roster.
- **Roster + structural summary for big config dicts.** `sorted(keys())` plus a
  per-entry `{type, len/value}` summary pins the whole `validParameters` shape
  and surfaces any added/removed/retyped parameter as a diff.

## Go / no-go on the next target

### Verdict: **GO — `SolutionStructs/Problem.py` ProblemType slice next** (defer `GlobalParameters.py`)

| Candidate | Why / why not | Effort to ≥95% line |
|---|---|---|
| **`SolutionStructs/Problem.py` ProblemType slice (~1382 LOC)** ✅ chosen | **Clean imports** — `rocisa.enum`, `DataType`, `ActivationType`, `Constants`, `Utilities`; **no** `isaInfoMap`/`asmCaps`/`assembler`/`subprocess`/toolchain references anywhere in the file. It is the natural sequel: `ProblemType` is exactly the object `LibraryIO` serialises and whose dtype fields `DataType` defines, so the type→validate→serialize arc (DataType → ValidParameters → **ProblemType** → LibraryIO ✓) closes here. Scope the slice to `ProblemType` (the `Mapping`), `problemTypeToEnum`, `validateProblemTypeParameterTypes`, `getRealDataTypeA/B`, and `ProblemSizeRange`/`ProblemSizes`/`ExactList`/`ExactDict` — reusing this effort's `DataType` knowledge and the LibraryIO vendored logic fixture for a realistic ProblemType. | ~1.5–2.5 days (slice) |
| `Common/GlobalParameters.py` (767 LOC) | High leverage but **heavily environment-coupled**: imports `subprocess`/`time`/`os.path`/`__version__`, mutates process-global `globalParameters` state (~144 refs), `setupRestoreClocks` shells out to set GPU clocks, and `assignGlobalParameters`/`printCapabilitiesTable` require an `isaInfoMap`. Testable, but monkeypatch-heavy and stateful — lower coverage-per-effort, and better tackled after the pure Problem core. | ~2–3 days (monkeypatch-heavy) |

**Why the Problem slice over GlobalParameters now:** it is the pure,
high-leverage continuation of the type/serialization arc and reuses everything
this effort has built (DataType pinned, ValidParameters pinned, the LibraryIO
logic fixture available for a real ProblemType). `GlobalParameters` is the more
environmental, stateful surface — defer it until the pure `Problem`/`Solution`
core is characterised, then give it the monkeypatch treatment LibraryIO's
`getCUCount` got.

### Effort estimate for the ProblemType slice

~1.5–2.5 days. `ProblemType` is a `Mapping` with non-trivial construction
(`FromOriginalState`-style defaulting, dtype-field derivation via `DataType`,
activation handling) — heavier than a flat predicate surface, but pure and
fixture-free for the core paths. Care items: build `ProblemType` from a minimal
real state dict (or reuse the LibraryIO logic fixture's ProblemType block);
snapshot a **normalised** view of the resulting Mapping (sorted items, dtype
fields as names) rather than the live object; parametrize
`validateProblemTypeParameterTypes` accept/reject and `problemTypeToEnum` across
dtype combos. A grounded API inventory + BEFORE baseline at kickoff (per the
established pattern) is in the companion `next-goal-problemtype.md`.
