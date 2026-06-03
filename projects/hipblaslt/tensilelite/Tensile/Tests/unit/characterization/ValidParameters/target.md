# Characterization target — `Tensile/Common/ValidParameters.py`

Follow-up to the completed `DataType/` suite (#7). Same pattern, next module.
Kept as a **new file** under the `ValidParameters/` test dir rather than editing
the shared `../target.md`, per the add-only rule. See `../survey.md` for the
syrupy survey (reused, no new survey needed).

## Module under test

One source file, `Tensile/Common/ValidParameters.py` (1136 LOC, 169 stmts). It
is the catalogue of legal Tensile solution-parameter values: a set of pure
`lru_cache`d table builders (`makeValidWorkGroups` / `makeValidWMMA` /
`makeValidSWMMAC` / `makeValidMFMA` / `makeValidSMFMA` /
`makeValidMatrixInstructions`), the giant declarative `validParameters` dict
(one import-time assignment spanning ~L228–1076, so the LOC dwarfs the
statement count), and three validators
(`checkSpaceFillAlgoIsValid` / `checkSpaceFillAlgoWGMIsValid` /
`checkParametersAreValid`). **Pure and self-contained** — imports only `math`,
`functools.lru_cache`, `.Architectures.SUPPORTED_ISA`, `.Types.IsaVersion`; no
toolchain, no subprocess, no `asmCaps`, no GPU.

| Area | LOC | Public API (tier) |
|---|---|---|
| `makeValidWorkGroups` | 82-93 | `lru_cache` list builder (A) |
| `makeValidWMMA` / `makeValidSWMMAC` | 94-99 | static list builders (A) |
| `makeValidMFMA` | 100-174 | `lru_cache` dtype-combo→MI dict + `_format9` expansion (A) |
| `makeValidSMFMA` | 175-206 | `lru_cache` sparse variant (A) |
| `makeValidMatrixInstructions` | 207-226 | `lru_cache` master MI list (A) |
| `validParameters` (dict) | 228-1076 | parameter catalogue, import-time (A) |
| `checkSpaceFillAlgoIsValid` | 1077-1090 | space-filling order validator (A) |
| `checkSpaceFillAlgoWGMIsValid` | 1091-1108 | WGM nested-pair validator (A) |
| `checkParametersAreValid` | 1109-1136 | central parameter validator (A) |
| **TOTAL** | | **169 stmts** |

## Why this module

- **Highest coverage-per-effort remaining, and pure.** Recommended by the
  `DataType` suite's `recommendations.md` ("GO → `Common/ValidParameters.py`
  next"). Same "static tables + pure validators, no environmental coupling"
  profile that let DataType hit 100% in hours, at larger scale.
- **Complements the `Validators/` suite.** Those tests check a *solution's*
  matrix instruction against the very tables this module builds; pinning the
  builders + the parameter validator anchors the legal-value vocabulary the
  rest of the toolchain validates against.

## Determinism handling

Every output is a pure function of the inputs — no paths, versions, timestamps,
or env coupling, so **no normalisation is required**. Technique notes:

1. **`lru_cache` builders.** Called and snapshotted directly; an identity test
   confirms the cache returns the same object on repeat calls.
2. **Large structures → normalised summaries.** `makeValidMatrixInstructions`
   returns ~746k entries and the `_format9` expansions are large; these are
   pinned by a deterministic `{len, head, tail}` (and per-key `{len, head}` for
   the MFMA/SMFMA dicts) rather than dumped whole, keeping the `.ambr`
   reviewable while still catching shape/content changes. Small lists
   (`WMMA`/`SWMMAC`) are snapshotted whole.
3. **`validParameters` table.** Pinned by `sorted(keys())` (the 138-name
   roster) plus a per-entry structural summary (type + length/value), so an
   added/removed/retyped parameter surfaces as a diff without dumping the
   multi-hundred-thousand-entry value lists.
4. **Validators via `pytest.raises`.** Accept paths assert a `None` return;
   each reject branch pins the raised message. Targeted synthetic `validParams`
   dicts drive every branch of `checkParametersAreValid` (early returns,
   unknown name, value-not-in-list with and without the `>32`-combos suffix,
   the `-1` any-value accept, and the `SpaceFillingAlgo`/`SFCWGM` sub-validator
   dispatch); the real `validParameters` drives one realistic accept.

## Location & coverage command (same rules as prior targets)

Suite at `Tensile/Tests/unit/characterization/ValidParameters/`, marked
`-m unit` (collected by the existing `testpaths=Tensile/Tests`, no config edit).
Pass `--cov` the package dir + grep the `Common/ValidParameters.py` row:

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "Common/ValidParameters.py"
```

Line coverage = `(Stmts - Miss) / Stmts` (the goal's bar is line; both line and
blended are reported).

## Result (before → after)

| | Stmts | Miss | Line cov | Blended |
|---|---|---|---|---|
| Before | 169 | 33 | 80.47% | 76.68% |
| After  | 169 | 0 | **100.00%** | 100.00% |

Delta: **+19.53 pts line**, −33 missing statements — **full line and branch
coverage** (0 missing, 0 partial branches). The existing suite already drove
the builders + the `validParameters` dict (they run during Solution/parameter
validation); the entire baseline gap was the three validators at the tail
(L1078–1136), which the new validator suite closes. **Nothing resisted** — see
`resistance.md`.

No regression: full `-m unit` went **1526 → 1563 passed** (+37 new tests),
201 skipped unchanged. Per-row detail in `coverage-after.txt`; next-target
go/no-go in `recommendations.md`.

### Suite layout (new files in this dir, add-only)

| File | Drives |
|---|---|
| `test_builders_char.py` | `makeValidWorkGroups`/`WMMA`/`SWMMAC`/`MFMA`/`SMFMA`/`MatrixInstructions` (summaries + `lru_cache` identity) and the `validParameters` roster + structural summary |
| `test_validators_char.py` | `checkSpaceFillAlgoIsValid`/`WGMIsValid` (accept + every reject branch) and `checkParametersAreValid` (early returns, accepts incl. `-1` sentinel, unknown-name, value-not-in short/long list, `SpaceFillingAlgo`/`SFCWGM` dispatch) |
