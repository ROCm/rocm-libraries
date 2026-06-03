# Characterization target — `Tensile/Common/DataType.py`

Follow-up to the completed `LibraryIO/` suite (#7). Same pattern, next module.
Kept as a **new file** under the `DataType/` test dir rather than editing the
shared `../target.md`, per the add-only rule. See `../survey.md` for the syrupy
survey (reused, no new survey needed).

## Module under test

One source file, `Tensile/Common/DataType.py` (493 LOC, 161 stmts). It defines
the `DataType` class — a thin wrapper over a static `properties` table (one
dict per dtype: `enum` / `char` / `nameAbbrev` / `miOutTypeNameAbbrev` / `reg`
/ `hip` / `isComplex`) — plus the module-level `_populateLookupTable` that
builds the `str -> row-index` lookup, and an `lru_cache`d `_is8bitFloat`
helper. Almost entirely **pure**: no filesystem, no toolchain, no GPU. The only
dependency is `rocisa.enum.DataTypeEnum` (already built in the env).

| Area | LOC | Public API (tier) |
|---|---|---|
| `__init__` (enum / int / str / DataType / invalid) | 272-284 | constructor, 4 forms + `RuntimeError` (A) |
| `toChar` / `toName` / `toNameAbbrev` / `toEnum` | 286-293 | table-read converters (A) |
| `toDevice(language)` | 294-298 | HIP name; non-HIP `assert 0` (A) |
| `zeroString(language, vectorWidth)` | 301-327 | string builder, vw 1 / >1 (A) |
| `is*` predicates (~43 methods) | 329-439 | pure booleans over the table (A) |
| `numRegisters` / `numBytes` / `MIOutputTypeNameAbbrev` / `flopsPerMac` | 441-448 | numeric/string reads (A) |
| `state` / `__str__` / `__repr__` / `getAttributes` | 450-458 | state forms (A) |
| `__hash__` / `__eq__` / `__lt__` | 460-473 | hashing + `total_ordering` (A) |
| `_populateLookupTable` | 477-490 | lookup builder + 2 guard raises (A/B) |
| **TOTAL** | | **161 stmts** |

## Why this module

- **Highest coverage-per-effort remaining.** Recommended by the `LibraryIO`
  suite's `recommendations.md` ("GO — `Common/DataType.py` next"). One class,
  one static table, a predicate-heavy pure surface — snapshots trivially with
  zero environmental coupling.
- **Foundational type.** Every other module (the `ProblemType` dtype fields the
  `LibraryIO` target just round-tripped, the `Solution`/`Problem` core ahead)
  is expressed in terms of `DataType`. Pinning its truth table now anchors the
  modules that build on it.

## Determinism handling

Every output is a pure function of the dtype and the call arguments — no paths,
versions, timestamps, or env coupling, so **no normalisation is required**.
Technique notes:

1. **Table-driven parametrization.** The dtype set is enumerated from
   `DataType.properties` (not hard-coded), so the matrix tracks any future
   table additions automatically.
2. **One snapshot per method, keyed by dtype name.** Each converter / predicate
   / numeric read is snapshotted as a `{dtype_name: result}` mapping (one
   `.ambr` entry per method) — the full truth table, kept reviewable.
3. **Introspected predicate roster.** The `is*` set is discovered via
   `inspect.getmembers` and pinned by its own snapshot, so adding/removing a
   predicate shows up as a diff rather than silently changing coverage.
4. **Raise paths via `pytest.raises`.** Invalid constructor input
   (`RuntimeError`), unknown lookup key (`KeyError`), non-HIP `toDevice`
   (`AssertionError`), `__lt__` vs non-DataType (`TypeError`), and the two
   `_populateLookupTable` guards (index-mismatch + duplicate-key, driven by
   tiny synthetic property lists) are all pinned by the raised type/message.

## Location & coverage command (same rules as prior targets)

Suite at `Tensile/Tests/unit/characterization/DataType/`, marked `-m unit`
(collected by the existing `testpaths=Tensile/Tests`, no config edit). Pass
`--cov` the package dir + grep the `Common/DataType.py` row (single-file
path-prefix does not match — see `coverage-before.txt`):

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "Common/DataType.py"
```

Line coverage = `(Stmts - Miss) / Stmts` (the goal's bar is line; both line and
blended are reported).

## Result (before → after)

| | Stmts | Miss | Line cov | Blended |
|---|---|---|---|---|
| Before | 161 | 41 | 74.53% | 74.05% |
| After  | 161 | 0 | **100.00%** | 100.00% |

Delta: **+25.47 pts line**, −41 missing statements — **full line and branch
coverage** (0 missing, 0 partial branches). The pre-existing suite already
drove DataType broadly but indirectly (Solution/Problem construction), so the
baseline was moderate; the new suite closes the predicate/converter/dunder
surface and the two `_populateLookupTable` guard raises. **Nothing resisted** —
see `resistance.md` for the (empty) resistance ledger and the notes on why the
nominally-defensive paths were all reachable here.

No regression: full `-m unit` went **1443 → 1526 passed** (+83 new tests),
201 skipped unchanged. Per-row detail in `coverage-after.txt`; next-target
go/no-go in `recommendations.md`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `test_constructor_char.py` | `__init__` (4 forms + invalid), `to*` converters, `toDevice` (HIP + assert), `zeroString` (vw 1/>1) |
| `test_predicates_char.py` | the introspected `is*` predicate matrix over every dtype + `_is8bitFloat` |
| `test_numeric_dunder_char.py` | `numRegisters`/`numBytes`/`flopsPerMac`/`MIOutputTypeNameAbbrev`, `state`/`__str__`/`__repr__`/`getAttributes`, `__hash__`/`__eq__`/`__lt__`, `_populateLookupTable` (+ both guards) |
