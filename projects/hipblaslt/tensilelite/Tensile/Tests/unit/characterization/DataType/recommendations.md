# Recommendations — after `Tensile/Common/DataType.py`

New file in the `DataType/` test dir per the add-only rule. Builds on the
`LibraryIO/` "GO → DataType next" verdict and the running cost model.

## Result for this target

`Tensile/Common/DataType.py` went from **74.53% line (161 stmts, 41 missing)**
to **100.00% line and branch (161 stmts, 0 missing, 0 partial)** — see
`coverage-after.txt`. **Nothing resisted**; the nominally-defensive raises, the
asserted non-HIP branch, and both `_populateLookupTable` guards were all
reachable (synthetic property lists for the two table guards; `pytest.raises`
for the rest). See `resistance.md`.

This came in well under the prior cost model's ~0.5–1 day estimate — closer to
a few hours. The module is exactly what the kickoff predicted: one class over a
static table, an introspectable pure predicate surface, no I/O, no toolchain,
no GPU. The whole suite is 3 files / 83 tests / 77 snapshots and runs in <0.5s
standalone.

No regression: full `-m unit` went **1443 → 1526 passed** (+83), 201 skipped
unchanged — purely additive.

## What worked (additions to the shared list)

- **Discover the axes from the module, pin them with their own snapshot.** Both
  the dtype set (iterate `DataType.properties`) and the `is*` predicate roster
  (`inspect.getmembers`) are derived at runtime and each pinned by a dedicated
  snapshot. The matrix then tracks the class automatically, and a future
  table/predicate change shows up as a snapshot diff instead of silently
  shifting coverage.
- **One snapshot per method, keyed by dtype name.** A `{dtype_name: result}`
  mapping per converter/predicate/numeric-read keeps each `.ambr` entry a
  reviewable truth table rather than 24× the test count.
- **Synthetic mini-tables for table-guard raises.** The index-mismatch and
  duplicate-key guards in `_populateLookupTable` are dead against the real
  (self-consistent) table; feeding 1–2 row synthetic `properties` lists reaches
  both without touching the module (add-only safe).
- **Pin hashing structurally, not by value.** Snapshot `equal⇒equal-hash`,
  `hash == hash(getAttributes())`, `distinct⇒distinct` rather than the raw int.

## Go / no-go on the next target

### Verdict: **GO — `Common/ValidParameters.py` next** (then a `Problem.py` ProblemType slice)

| Candidate | Why / why not | Effort to ≥95% line |
|---|---|---|
| **`Common/ValidParameters.py` (1136 LOC)** ✅ chosen | **Pure and self-contained** — imports only `math`, `functools`, `SUPPORTED_ISA`, `IsaVersion` (no toolchain, no subprocess, no `asmCaps`, no GPU). It is a set of `makeValid*` table builders (`makeValidWorkGroups`/`WMMA`/`SWMMAC`/`MFMA`/`SMFMA`/`MatrixInstructions`) returning static structures, a big declarative `validParameters` dict, and the pure validators `checkParametersAreValid` / `checkSpaceFillAlgoIsValid` / `checkSpaceFillAlgoWGMIsValid`. Snapshot the built tables; parametrize the validators' accept/reject paths. Same profile that made DataType cheap, at larger scale. Also closes the loop with the `Validators/` suite (which validates a *solution's* MI against these very tables). | ~1–1.5 days |
| `SolutionStructs/Problem.py` ProblemType slice (~1382 LOC) | The natural sequel to LibraryIO (it builds the `ProblemType` that LibraryIO serialises and that DataType's fields populate). But `ProblemType` is a `Mapping` with heavy `FromOriginalState`-style construction entangled with `Solution`; needs the real `isa_info_map`/`assembler` fixtures. Scope to `ProblemType` + `problemTypeToEnum` + `validateProblemTypeParameterTypes` + `ProblemSizeRange`/`ProblemSizes` as a **slice**, reusing the LibraryIO suite's fixtures and vendored logic fixture. | ~1.5–2.5 days (slice) |
| `Common/GlobalParameters.py` (767 LOC) | Process/env-global state (defaults, assignments, version), more environmental coupling and mutable globals — lower leverage, defer. | ~2 days |

**Why `ValidParameters` over the `Problem` slice now:** it is the cheapest
remaining high-leverage *pure* module — the same "static tables + pure
validators, no environmental coupling" shape that let DataType hit 100% in
hours, just bigger. Banking it next continues the established
cheapest-pure-first strategy and directly complements the `Validators/` suite
(those tests check a solution against the tables `ValidParameters` builds).
Take the `Problem.py` `ProblemType` slice immediately after — by then DataType
+ ValidParameters give the type and legal-value vocabulary that a ProblemType
characterization leans on, and the LibraryIO fixtures are reused for the heavy
construction.

### Effort estimate for `ValidParameters`

~1–1.5 days. No I/O, no GPU, but larger and more combinatorial than DataType:
the `makeValid*` builders produce sizeable nested structures (snapshot them
whole, or a normalised summary — counts + a sorted sample — if an `.ambr` would
be unwieldy), and `checkParametersAreValid` has several reject branches
(unknown name, out-of-range value, the `>32`-combos message variant, the
`SpaceFillingAlgo`/`SFCWGM` sub-validators) to parametrize. A grounded API
inventory + BEFORE baseline at kickoff (per the established pattern) is in the
companion `next-goal-validparameters.md`.
