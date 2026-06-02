# Recommendations — after `Tensile/LibraryIO.py`

New file in the `LibraryIO/` test dir per the add-only rule. Builds on the
`TensileLogic/` "GO" verdict and cost model.

## Result for this target

`Tensile/LibraryIO.py` went from **32.81% line (448 stmts, 301 missing)** to
**98.21% line (448 stmts, 8 missing; +65.40 pts)** — see `coverage-after.txt`.
The 8 residual line-misses are all provably unreachable here (absent optional
packages, the mandatory-yaml `printExit`, and redundant per-solution
re-defaulting); details in `resistance.md`. No regression: full `-m unit`
went **1330 → 1443 passed** (+113), 201 skipped unchanged — purely additive.

This landed within the prior cost model's `LibraryIO` estimate (~1–2 days). The
pure half (serialisers, dtype mappers, `LibraryLogic`, `read*`/`write*`) was
near-free; the contract half (`parseLibraryLogic*`, `parseSolutions*`,
`createLibraryLogic`) needed a real fixture + a genuine write→read round-trip,
exactly as the kickoff predicted.

## What worked (additions to the shared list)

- **Vendor one real fixture, round-trip everything else.** A single verbatim
  production logic file (`data/logic_gfx942_HSS_BH.yaml`) drives the entire
  heavy parse surface; the solutions-file path is then generated *from* the
  parsed solutions (`writeSolutions` → `parseSolutionsFile`) rather than
  hand-authored. One real input, both contracts pinned.
- **Snapshot a normalised structural summary, not live objects.** Parsed
  results hold `Solution`/`ProblemType`/`MasterSolutionLibrary` instances;
  snapshotting `{schedule, arch, counts, sorted type-mismatches, selected PT
  fields}` keeps them deterministic and review-friendly.
- **Reload-with-blocked-deps for import fallbacks.** `except ImportError` arms
  that never run when the fast deps are installed are covered by
  `importlib.reload` under `sys.modules[x]=None` / `delattr`, restored in a
  `finally`. This is the only way to reach env-gated import code without
  editing the module (add-only) and is safe for the shared session (verified).
- **`ValueError`-before-construction trick.** The custom-kernel bad-MI branch
  raises before any `Solution` is built, so a length-3 `MatrixInstruction`
  config covers it without needing a self-consistent kernel.

## Go / no-go on the next target

### Verdict: **GO — `Common/DataType.py` next** (then a `Problem.py` slice)

| Candidate | Why / why not | Effort to ≥95% line |
|---|---|---|
| **`Common/DataType.py` (~492 LOC)** ✅ chosen | One class wrapping a static `properties` table, with ~80 **pure** predicate methods (`isHalf`/`isAnyFloat8`/`is8bitFloat`/…) and converters (`toChar`/`toName`/`toEnum`/`toDevice`/`zeroString`). Parametrise over *all* dtypes × methods and snapshot — almost entirely Tier A, no fixtures, no toolchain, no GPU. Highest coverage-per-effort remaining. | ~0.5–1 day |
| `Common/ValidParameters.py` (~1136 LOC) | High leverage but large; mostly a big declarative table + validators. Good *after* DataType. | ~2–3 days |
| `SolutionStructs/Problem.py` slice (~1382 LOC) | Natural sequel to LibraryIO (it constructs the `ProblemType` LibraryIO serialises). But large and entangled with `Solution`; scope to `ProblemType` + `problemTypeToEnum` + `ProblemSizes` as a **slice**, reusing this suite's `isa_info_map`/`assembler` fixtures and the vendored logic fixture. | ~2–4 days (full); ~1–1.5 days (slice) |

**Why `DataType` over a `Problem` slice now:** `DataType` is the cheapest,
most self-contained high-leverage module left — everything (including the
`ProblemType` dtype fields this target just round-tripped) is expressed in
terms of it, and its pure predicate surface snapshots trivially with zero
environmental coupling. Bank that near-free coverage first, then take the
`Problem.py` `ProblemType` slice (which can reuse this suite's fixtures) as the
next step toward the Solution/Problem core. Defer the large
`ValidParameters`/`Solution` surfaces.

### Effort estimate for `DataType`

~0.5–1 day. Single file, single class, no I/O. The only mild care items:
`toDevice(language)`/`zeroString(language, vectorWidth)` branch on language
strings (parametrise `HIP`/`HLSL`/etc.), and any `properties`-lookup paths for
dtypes not present in the table (snapshot the raised error). A grounded API
inventory + BEFORE baseline at kickoff (per the established pattern) is in the
companion `next-goal-datatype.md`.
